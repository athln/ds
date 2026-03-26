"""
ann_gui.py  —  Multi-Model ANN Predictor  (Light Theme)
=========================================================
Loads five trained ANN checkpoints (ann_model_1.pt ... ann_model_5.pt) and
predicts five damage-state outputs (DS-1 ... DS-5) from the same 8 structural
inputs in a single clean, light-mode interface.

Input features (fixed):
    fc | fy | Average Mass Loss | Pitting Mass Loss | Transverse Mass Loss
    Story Height | Bay Width | Slab Thickness

Output targets:
    DS-1  DS-2  DS-3  DS-4  DS-5

Usage:
    python ann_gui.py                      # auto-finds models next to this file
    python ann_gui.py --model_dir /path/to/folder
"""

import argparse, pickle, pathlib, sys, csv, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DOMAIN CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    "Concrete Compressive Strength",
    "Steel Bar Yield Strength",
    "Average Mass Loss",
    "Pitting Mass Loss",
    "Transverse Mass Loss",
    "Story Height",
    "Bay Width",
    "Slab Thickness",
]
N_FEATURES  = len(FEATURE_NAMES)          # 8
OUTPUT_NAMES = ["DS-1", "DS-2", "DS-3", "DS-4", "DS-5"]
N_OUTPUTS   = len(OUTPUT_NAMES)           # 5
MODEL_FILES = [f"ann_model_{i+1}.pt" for i in range(N_OUTPUTS)]

# Conversion factor: fc and fy are entered by the user in ksi,
# but the ANN was trained with values in MPa.
# Before inference, both are multiplied by this factor (1 ksi = 6.89476 MPa).
KSI_TO_MPA = 0.145
FC_FY_INDICES = [
    FEATURE_NAMES.index("Concrete Compressive Strength"),
    FEATURE_NAMES.index("Steel Bar Yield Strength"),
]

FIELD_HINTS = {
    "Concrete Compressive Strength":   ("MPa", "20 – 30"),
    "Steel Bar Yield Strength":     ("MPa", "400 – 530"),
    "Average Mass Loss":    ("%",              "0 – 30"),
    "Pitting Mass Loss":    ("%",              "0 – 50"),
    "Transverse Mass Loss": ("%",              "0 – 70"),
    "Story Height":         ("m",              "3 – 4"),
    "Bay Width":            ("m",              "3 – 5"),
    "Slab Thickness":       ("m",             "0.12 – 0.2"),
}

# ── Light-mode palette ───────────────────────────────────────────────────────
BG          = "#f0f4f8"          # page background — cool off-white
BG2         = "#ffffff"          # panel / card surface
BG3         = "#e8edf2"          # alternate row / subtle card
BG4         = "#f8fafc"          # input field fill
HDR_BG      = "#1c3d6e"          # deep navy header bar
HDR_BG2     = "#2a5298"          # slightly lighter header accent
ACCENT      = "#1971c2"          # primary blue
ACCENT2     = "#1864ab"          # darker blue for headings
ACCENT_DARK = "#155a9c"          # hover / pressed
TEXT        = "#1a202c"          # near-black main text
TEXT_DIM    = "#64748b"          # muted secondary text
TEXT_LIGHT  = "#94a3b8"          # placeholder / hint
BORDER      = "#cbd5e1"          # light border
BORDER2     = "#e2e8f0"          # very subtle separator
SUCCESS     = "#276749"          # dark green — metrics / positive
WARNING_COL = "#92400e"          # amber warning
ERROR_COL   = "#9b1c1c"          # red error

# Per-DS colours — vivid but readable on white
DS_COLORS = [
    "#1d6f42",   # DS-1  forest green
    "#1564ad",   # DS-2  strong blue
    "#b45309",   # DS-3  amber / burnt orange
    "#9b1c1c",   # DS-4  crimson
    "#5b21b6",   # DS-5  indigo
]

# DS fill tints (very light background for cards)
DS_TINTS = [
    "#ecfdf5",   # DS-1 mint
    "#eff6ff",   # DS-2 sky
    "#fffbeb",   # DS-3 cream
    "#fef2f2",   # DS-4 rose
    "#f5f3ff",   # DS-5 lavender
]

# ── Typography ───────────────────────────────────────────────────────────────
FT        = ("Segoe UI", 10)
FT_BOLD   = ("Segoe UI", 10, "bold")
FT_TITLE  = ("Segoe UI", 17, "bold")
FT_HEAD   = ("Segoe UI", 11, "bold")
FT_SMALL  = ("Segoe UI", 9)
FT_MONO   = ("Consolas", 10)
FT_DS_NUM = ("Segoe UI", 22, "bold")
FT_DS_LBL = ("Segoe UI", 8, "bold")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL ARCHITECTURE  (must match training code)
# ══════════════════════════════════════════════════════════════════════════════
class ANN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CHECKPOINT LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_checkpoint(path):
    ckpt     = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg      = ckpt["model_config"]
    model    = ANN(cfg["input_dim"], cfg["hidden_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler_X = pickle.loads(ckpt["scaler_X"])
    scaler_y = pickle.loads(ckpt["scaler_y"])
    meta     = {
        "cv_metrics"  : ckpt.get("cv_metrics",   {}),
        "train_cfg"   : ckpt.get("train_cfg",    {}),
        "model_config": cfg,
    }
    return model, scaler_X, scaler_y, meta

# ══════════════════════════════════════════════════════════════════════════════
# 4.  PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def predict_one(model, sx, sy, values):
    arr = np.array(values, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        out = model(torch.tensor(sx.transform(arr))).numpy()
    return float(sy.inverse_transform(out)[0, 0])

def predict_all_single(bundles, values):
    return [predict_one(m, sx, sy, values) for m, sx, sy, _ in bundles]

def predict_all_batch(bundles, X_arr):
    results = []
    for model, sx, sy, _ in bundles:
        X_sc = sx.transform(X_arr.astype(np.float32))
        with torch.no_grad():
            p = model(torch.tensor(X_sc)).numpy()
        results.append(sy.inverse_transform(p).ravel())
    return np.column_stack(results)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class ANNPredictorApp(tk.Tk):

    def __init__(self, model_dir: pathlib.Path):
        super().__init__()
        self.title("ANN Damage State Predictor v1.0")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1040, 700)

        # ── load all 5 models ──────────────────────────────────
        self.bundles = []
        missing = []
        for fname in MODEL_FILES:
            p = model_dir / fname
            if not p.exists():
                missing.append(str(p))
            else:
                try:
                    self.bundles.append(load_checkpoint(p))
                except Exception as e:
                    messagebox.showerror("Load Error",
                        f"Failed to load {fname}:\n{e}")
                    self.destroy(); return

        if missing:
            ans = messagebox.askyesno(
                "Missing Models",
                "The following model file(s) were not found:\n\n"
                + "\n".join(missing)
                + "\n\nProceed with the models that were found?")
            if not ans:
                self.destroy(); return

        if not self.bundles:
            messagebox.showerror("Error", "No models could be loaded.")
            self.destroy(); return

        self.loaded_outputs = OUTPUT_NAMES[:len(self.bundles)]
        self.loaded_colors  = DS_COLORS[:len(self.bundles)]
        self.loaded_tints   = DS_TINTS[:len(self.bundles)]

        self._apply_style()
        self._build_ui()
        self.update_idletasks()
        self._center_window()

    # ──────────────────────────────────────────────────────────
    # STYLE
    # ──────────────────────────────────────────────────────────
    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        # global defaults
        s.configure(".",
                    background=BG, foreground=TEXT,
                    font=FT, borderwidth=0,
                    troughcolor=BORDER)

        # frames
        s.configure("TFrame",     background=BG)
        s.configure("White.TFrame", background=BG2)

        # labels
        s.configure("TLabel",     background=BG,  foreground=TEXT)
        s.configure("Dim.TLabel", background=BG,  foreground=TEXT_DIM)

        # notebook
        s.configure("TNotebook",
                    background=BG, borderwidth=0, tabmargins=0)
        s.configure("TNotebook.Tab",
                    background=BG3, foreground=TEXT_DIM,
                    padding=[18, 8], font=FT)
        s.map("TNotebook.Tab",
              background=[("selected", BG2)],
              foreground=[("selected", ACCENT2)])

        # primary button
        s.configure("Accent.TButton",
                    background=ACCENT, foreground="#ffffff",
                    font=FT_BOLD, padding=[14, 8],
                    relief="flat", borderwidth=0)
        s.map("Accent.TButton",
              background=[("active",  HDR_BG2),
                          ("pressed", ACCENT_DARK)])

        # ghost / secondary button
        s.configure("Ghost.TButton",
                    background=BG3, foreground=TEXT_DIM,
                    font=FT, padding=[10, 6], relief="flat",
                    borderwidth=1)
        s.map("Ghost.TButton",
              background=[("active", BORDER)],
              foreground=[("active", TEXT)])

        # scrollbar
        s.configure("TScrollbar",
                    background=BG3, troughcolor=BORDER2,
                    arrowcolor=TEXT_DIM, borderwidth=0,
                    relief="flat")

        # treeview
        s.configure("Treeview",
                    background=BG2, foreground=TEXT,
                    fieldbackground=BG2, rowheight=28,
                    font=FT_SMALL, borderwidth=1,
                    relief="solid")
        s.configure("Treeview.Heading",
                    background=BG3, foreground=ACCENT2,
                    font=("Segoe UI", 9, "bold"),
                    relief="flat")
        s.map("Treeview",
              background=[("selected", "#dbeafe")],
              foreground=[("selected", ACCENT2)])

        # separator
        s.configure("TSeparator", background=BORDER)

    # ──────────────────────────────────────────────────────────
    # TOP-LEVEL LAYOUT
    # ──────────────────────────────────────────────────────────
    def _build_ui(self):

        # ── navy header bar ────────────────────────────────────
        hdr = tk.Frame(self, bg=HDR_BG, pady=0)
        hdr.pack(fill="x")

        # left: logo + title
        left_hdr = tk.Frame(hdr, bg=HDR_BG, padx=20, pady=14)
        left_hdr.pack(side="left")
        tk.Label(left_hdr, text="\u29c6",
                 font=("Segoe UI", 20, "bold"),
                 bg=HDR_BG, fg="#93c5fd").pack(side="left", padx=(0, 10))
        tk.Label(left_hdr, text="Damage State Predictor",
                 font=FT_TITLE, bg=HDR_BG, fg="#ffffff").pack(side="left")

        # right: status chips
        right_hdr = tk.Frame(hdr, bg=HDR_BG, padx=20)
        right_hdr.pack(side="right", fill="y")

        status_txt = (
            f"{N_FEATURES} Inputs   \u2192   "
            f"{len(self.bundles)} Damage States   \u2192   "
            + "  |  ".join(self.loaded_outputs)
        )
        tk.Label(right_hdr, text=status_txt,
                 font=("Segoe UI", 9), bg=HDR_BG,
                 fg="#93c5fd").pack(side="right", anchor="center")

        # ── thin accent stripe below header ───────────────────
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")

        # ── notebook ──────────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        t1 = ttk.Frame(nb, style="TFrame")
        t2 = ttk.Frame(nb, style="TFrame")
        t3 = ttk.Frame(nb, style="TFrame")

        nb.add(t1, text="   \u26a1  Single Prediction   ")
        nb.add(t2, text="   \U0001f4cb  Batch Prediction    ")
        nb.add(t3, text="   \u2139\ufe0f  Model Info          ")

        self._build_single_tab(t1)
        self._build_batch_tab(t2)
        self._build_info_tab(t3)

    # ══════════════════════════════════════════════════════════
    # TAB 1 — SINGLE PREDICTION
    # ══════════════════════════════════════════════════════════
    def _build_single_tab(self, parent):
        parent.columnconfigure(0, weight=0, minsize=330)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        # ── LEFT: input panel ─────────────────────────────────
        left_wrap = tk.Frame(parent, bg=BG, padx=10, pady=10)
        left_wrap.grid(row=0, column=0, sticky="nsew")

        left = tk.Frame(left_wrap, bg=BG2,
                        highlightthickness=1,
                        highlightbackground=BORDER,
                        padx=22, pady=18)
        left.pack(fill="both", expand=True)
        left.columnconfigure(1, weight=1)

        # panel header
        hdr_row = tk.Frame(left, bg=BG2)
        hdr_row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 16))
        tk.Label(hdr_row, text="Input Parameters",
                 font=FT_HEAD, bg=BG2, fg=ACCENT2).pack(side="left")

        self.entries       = {}
        self.entry_widgets = {}

        for i, name in enumerate(FEATURE_NAMES):
            unit, hint = FIELD_HINTS.get(name, ("", ""))

            # row shading — alternating very subtle tint
            row_bg = BG2 if i % 2 == 0 else BG3

            tk.Label(left, text=name,
                     font=FT_SMALL, bg=BG2, fg=TEXT,
                     anchor="e").grid(
                row=i+1, column=0, sticky="e",
                padx=(0, 10), pady=6)

            var = tk.StringVar()
            self.entries[name] = var

            e = tk.Entry(left, textvariable=var, width=14,
                         font=FT_MONO,
                         bg=BG4, fg=TEXT,
                         insertbackground=ACCENT,
                         relief="solid", bd=1,
                         highlightthickness=1,
                         highlightbackground=BORDER,
                         highlightcolor=ACCENT)
            e.grid(row=i+1, column=1, sticky="ew", pady=6)
            e.bind("<Return>",   lambda ev: self._do_predict())
            e.bind("<FocusIn>",  lambda ev, w=e: w.config(
                highlightbackground=ACCENT, bg="#eff6ff"))
            e.bind("<FocusOut>", lambda ev, w=e: w.config(
                highlightbackground=BORDER, bg=BG4))
            self.entry_widgets[name] = e

            # unit + range hint
            hint_txt = f"{unit}" + (f"  [{hint}]" if hint else "")
            tk.Label(left, text=hint_txt,
                     font=("Segoe UI", 8), bg=BG2,
                     fg=TEXT_LIGHT, anchor="w").grid(
                row=i+1, column=2, sticky="w", padx=(8, 0), pady=6)

        # divider
        tk.Frame(left, bg=BORDER2, height=1).grid(
            row=N_FEATURES+1, column=0, columnspan=3,
            sticky="ew", pady=(14, 10))

        # buttons
        btn_row = tk.Frame(left, bg=BG2)
        btn_row.grid(row=N_FEATURES+2, column=0, columnspan=3, sticky="w")

        ttk.Button(btn_row, text="\u26a1  Predict All",
                   style="Accent.TButton",
                   command=self._do_predict).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Clear All",
                   style="Ghost.TButton",
                   command=self._clear_inputs).pack(side="left")

        self.single_status = tk.StringVar(
            value="Enter all 8 parameters and press Predict All")
        tk.Label(left, textvariable=self.single_status,
                 font=("Segoe UI", 8), bg=BG2, fg=TEXT_DIM,
                 wraplength=280, justify="left").grid(
            row=N_FEATURES+3, column=0, columnspan=3,
            sticky="w", pady=(10, 0))

        # ── RIGHT: results panel ───────────────────────────────
        right_wrap = tk.Frame(parent, bg=BG, padx=(0), pady=10)
        right_wrap.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        right_wrap.columnconfigure(0, weight=1)
        right_wrap.rowconfigure(1, weight=0)

        tk.Label(right_wrap, text="Predicted Damage States",
                 font=FT_HEAD, bg=BG, fg=ACCENT2).pack(
            anchor="w", pady=(0, 10))

        # 5 DS cards — 3 on row 0, 2 on row 1
        cards_frame = tk.Frame(right_wrap, bg=BG)
        cards_frame.pack(fill="x")
        cards_frame.columnconfigure(0, weight=1)
        cards_frame.columnconfigure(1, weight=1)
        cards_frame.columnconfigure(2, weight=1)

        self.result_vars  = {}
        self.result_cards = {}

        layout = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

        for ci, ds in enumerate(self.loaded_outputs):
            row_i, col_i = layout[ci]
            color  = self.loaded_colors[ci]
            tint   = self.loaded_tints[ci]

            card = tk.Frame(cards_frame, bg=BG2,
                            highlightthickness=1,
                            highlightbackground=BORDER)
            card.grid(row=row_i, column=col_i,
                      padx=5, pady=5, sticky="nsew")

            # coloured left accent bar
            accent_bar = tk.Frame(card, bg=color, width=5)
            accent_bar.pack(side="left", fill="y")

            body = tk.Frame(card, bg=tint, padx=14, pady=14)
            body.pack(side="left", fill="both", expand=True)

            tk.Label(body, text=ds,
                     font=FT_DS_LBL, bg=tint,
                     fg=color).pack(anchor="w")

            var = tk.StringVar(value="\u2014")
            self.result_vars[ds] = var
            tk.Label(body, textvariable=var,
                     font=FT_DS_NUM, bg=tint,
                     fg=color).pack(anchor="w", pady=(4, 0))

            self.result_cards[ds] = card

        # ── Author / version info box ──────────────────────────
        info_box = tk.Frame(right_wrap, bg=HDR_BG,
                            highlightthickness=1,
                            highlightbackground=ACCENT_DARK)
        info_box.pack(fill="x", pady=(14, 0))

        # left: avatar initials badge
        badge = tk.Frame(info_box, bg=ACCENT, width=84, height=54)
        badge.pack(side="left", padx=(14, 12), pady=12)
        badge.pack_propagate(False)
        tk.Label(badge, text="DS-ANN",
                 font=("Segoe UI", 14, "bold"),
                 bg=ACCENT, fg="#ffffff").place(relx=0.5, rely=0.5,
                                                anchor="center")

        # right: name + version text
        text_side = tk.Frame(info_box, bg=HDR_BG)
        text_side.pack(side="left", pady=12)

        tk.Label(text_side, text="ANN-Based Damage State Prediction for Corroded RC Frames",
                 font=("Segoe UI", 13, "bold"),
                 bg=HDR_BG, fg="#ffffff").pack(anchor="w")
        tk.Label(text_side, text="Version 1.0  |  DOI: ",
                 font=("Segoe UI", 9),
                 bg=HDR_BG, fg="#93c5fd").pack(anchor="w", pady=(2, 0))

    # ──────────────────────────────────────────────────────────
    def _do_predict(self):
        try:
            vals = [float(self.entries[n].get()) for n in FEATURE_NAMES]
        except ValueError:
            self.single_status.set(
                "\u26a0  All 8 fields must contain valid numbers.")
            for ds in self.loaded_outputs:
                self.result_vars[ds].set("\u2014")
            return

        # Convert fc and fy from ksi (user input) to MPa (model expects)
        for idx in FC_FY_INDICES:
            vals[idx] = vals[idx] * KSI_TO_MPA

        try:
            preds = predict_all_single(self.bundles, vals)
        except Exception as exc:
            self.single_status.set(f"Prediction error: {exc}")
            return

        for ds, val in zip(self.loaded_outputs, preds):
            self.result_vars[ds].set(f"{val:.4f}")

        self.single_status.set(
            "\u2713  Prediction complete  \u2014  "
            + "   ".join(f"{ds}: {v:.4f}"
                         for ds, v in zip(self.loaded_outputs, preds)))

    def _clear_inputs(self):
        for var in self.entries.values():
            var.set("")
        for ds in self.loaded_outputs:
            self.result_vars[ds].set("\u2014")
        self.single_status.set(
            "Enter all 8 parameters and press Predict All")

    # ══════════════════════════════════════════════════════════
    # TAB 2 — BATCH PREDICTION
    # ══════════════════════════════════════════════════════════
    def _build_batch_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # toolbar
        bar = tk.Frame(parent, bg=BG2,
                       highlightthickness=1,
                       highlightbackground=BORDER2,
                       padx=12, pady=10)
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        ttk.Button(bar, text="\U0001f4c2  Import CSV",
                   style="Accent.TButton",
                   command=self._import_csv).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="\U0001f4be  Export Results",
                   style="Ghost.TButton",
                   command=self._export_csv).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Clear",
                   style="Ghost.TButton",
                   command=self._clear_batch).pack(side="left")

        self.batch_status = tk.StringVar(
            value=f"Import a CSV with {N_FEATURES} columns "
                  f"({', '.join(FEATURE_NAMES[:3])} ...)")
        tk.Label(bar, textvariable=self.batch_status,
                 font=FT_SMALL, bg=BG2, fg=TEXT_DIM).pack(
            side="right", padx=10)

        # treeview
        all_cols  = FEATURE_NAMES + self.loaded_outputs
        tf = tk.Frame(parent, bg=BG)
        tf.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(tf, columns=all_cols,
                                  show="headings",
                                  selectmode="extended")
        for col in all_cols:
            is_out = col in self.loaded_outputs
            self.tree.heading(col, text=col)
            self.tree.column(col,
                             width=120 if is_out else 108,
                             minwidth=60, anchor="center")

        vsb = ttk.Scrollbar(tf, orient="vertical",
                             command=self.tree.yview)
        hsb = ttk.Scrollbar(tf, orient="horizontal",
                             command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set,
                             xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.tree.tag_configure("even", background="#ffffff")
        self.tree.tag_configure("odd",  background="#f8fafc")

        self._batch_data = None

    def _import_csv(self):
        path = filedialog.askopenfilename(
            title="Open Input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            rows = []
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None)       # skip header
                for row in reader:
                    if row:
                        rows.append([float(v) for v in row[:N_FEATURES]])
            if not rows:
                raise ValueError("No data rows found.")
            X_arr = np.array(rows, dtype=np.float32)
            if X_arr.shape[1] != N_FEATURES:
                raise ValueError(
                    f"Expected {N_FEATURES} columns, got {X_arr.shape[1]}.")

            # Convert fc and fy columns from ksi to MPa
            X_arr[:, FC_FY_INDICES] = X_arr[:, FC_FY_INDICES] * KSI_TO_MPA

            preds = predict_all_batch(self.bundles, X_arr)
            self._batch_data = (X_arr, preds)
            self._populate_tree(X_arr, preds)
            n = len(rows)
            self.batch_status.set(
                f"\u2713  {n} rows loaded from "
                f"{pathlib.Path(path).name}  \u2014  "
                f"{n} predictions computed")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))

    def _populate_tree(self, X_arr, preds):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, (xrow, prow) in enumerate(zip(X_arr, preds)):
            vals = ([f"{v:.5g}" for v in xrow]
                    + [f"{p:.5f}" for p in prow])
            tag  = "even" if i % 2 == 0 else "odd"
            self.tree.insert("", "end", values=vals, tags=(tag,))

    def _export_csv(self):
        if self._batch_data is None:
            messagebox.showinfo("No Data",
                                "Import a CSV and run predictions first.")
            return
        X_arr, preds = self._batch_data
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Results As ...")
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_NAMES + self.loaded_outputs)
            for xrow, prow in zip(X_arr, preds):
                writer.writerow(list(xrow) + list(prow))
        messagebox.showinfo("Exported", f"Results saved to:\n{path}")

    def _clear_batch(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._batch_data = None
        self.batch_status.set(
            f"Import a CSV with {N_FEATURES} columns "
            f"({', '.join(FEATURE_NAMES[:3])} ...)")

    # ══════════════════════════════════════════════════════════
    # TAB 3 — MODEL INFO
    # ══════════════════════════════════════════════════════════
    def _build_info_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # ── top: input features banner ─────────────────────────
        feat_wrap = tk.Frame(parent, bg=BG, padx=10, pady=(10))
        feat_wrap.grid(row=0, column=0, sticky="ew")

        feat_card = tk.Frame(feat_wrap, bg=BG2,
                             highlightthickness=1,
                             highlightbackground=BORDER,
                             padx=18, pady=12)
        feat_card.pack(fill="x")

        tk.Label(feat_card,
                 text="Input Features  (\u2192 shared by all 5 models)",
                 font=FT_HEAD, bg=BG2, fg=ACCENT2).pack(
            anchor="w", pady=(0, 10))

        feat_grid = tk.Frame(feat_card, bg=BG2)
        feat_grid.pack(fill="x")

        for i, name in enumerate(FEATURE_NAMES):
            unit = FIELD_HINTS.get(name, ("", ""))[0]
            col  = i % 4
            row  = i // 4

            chip = tk.Frame(feat_grid, bg=BG3,
                            highlightthickness=1,
                            highlightbackground=BORDER,
                            padx=10, pady=5)
            chip.grid(row=row, column=col, padx=5, pady=4, sticky="w")

            tk.Label(chip,
                     text=f"{i+1}.  {name}",
                     font=FT_SMALL, bg=BG3, fg=TEXT).pack(side="left")
            if unit:
                tk.Label(chip, text=f"  [{unit}]",
                         font=("Segoe UI", 8), bg=BG3,
                         fg=TEXT_DIM).pack(side="left")

        # ── bottom: scrollable model cards ────────────────────
        cf = tk.Frame(parent, bg=BG)
        cf.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)

        canvas = tk.Canvas(cf, bg=BG, highlightthickness=0)
        vsb    = ttk.Scrollbar(cf, orient="vertical",
                                command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        inner = tk.Frame(canvas, bg=BG)
        win   = canvas.create_window((0, 0), window=inner, anchor="nw")

        canvas.bind("<Configure>",
                    lambda ev: canvas.itemconfig(win, width=ev.width))
        canvas.bind_all("<MouseWheel>",
                        lambda ev: canvas.yview_scroll(
                            int(-1 * (ev.delta / 120)), "units"))
        inner.bind("<Configure>",
                   lambda ev: canvas.configure(
                       scrollregion=canvas.bbox("all")))

        for ci in range(len(self.bundles)):
            inner.columnconfigure(ci, weight=1)

        for ci, (ds, color, tint, (model, sx, sy, meta)) in enumerate(
                zip(self.loaded_outputs,
                    self.loaded_colors,
                    self.loaded_tints,
                    self.bundles)):

            card = tk.Frame(inner, bg=BG2,
                            highlightthickness=1,
                            highlightbackground=color)
            card.grid(row=0, column=ci, padx=6, pady=6, sticky="nsew")

            # top colour stripe
            tk.Frame(card, bg=color, height=6).pack(fill="x")

            body = tk.Frame(card, bg=BG2, padx=14, pady=12)
            body.pack(fill="both", expand=True)

            tk.Label(body, text=ds,
                     font=("Segoe UI", 15, "bold"),
                     bg=BG2, fg=color).pack(anchor="w", pady=(0, 10))

            cfg  = meta["model_config"]
            hl   = cfg["hidden_layers"]
            cv   = meta.get("cv_metrics", {})
            tcfg = meta.get("train_cfg",  {})

            _section_light(body, "Architecture")
            for k, v in [
                ("Input dim",     str(cfg["input_dim"])),
                ("Hidden layers", str(len(hl))),
                ("Neurons",       " \u2192 ".join(str(h) for h in hl)),
                ("Output dim",    "1"),
                ("Activation",    "ReLU"),
            ]:
                _kv_light(body, k, v, color)

            _section_light(body, "5-Fold CV Metrics")
            for label, key in [
                ("R\u00b2",  "r2"),
                ("RMSE",     "rmse"),
                ("NRMSE",    "nrmse"),
            ]:
                val = cv.get(key)
                txt = f"{val:.6f}" if isinstance(val, float) else "N/A"
                col_val = SUCCESS if isinstance(val, float) else TEXT_DIM
                _kv_light(body, label, txt, col_val)

            _section_light(body, "Training Config")
            for k, v in tcfg.items():
                _kv_light(body, k, str(v), TEXT)

        parent.update_idletasks()

    # ──────────────────────────────────────────────────────────
    def _center_window(self):
        w, h = 1120, 750
        sw   = self.winfo_screenwidth()
        sh   = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _section_light(parent, title):
    """Draws a labelled divider for model-info cards."""
    sep = tk.Frame(parent, bg=BORDER2, height=1)
    sep.pack(fill="x", pady=(10, 3))
    tk.Label(parent, text=title.upper(),
             font=("Segoe UI", 7, "bold"),
             bg=BG2, fg=TEXT_LIGHT).pack(anchor="w")


def _kv_light(parent, key, val, val_color):
    """Key/value row for model-info cards."""
    row = tk.Frame(parent, bg=BG2)
    row.pack(fill="x", pady=1)
    tk.Label(row, text=key,
             font=FT_SMALL, bg=BG2,
             fg=TEXT_DIM, width=15, anchor="w").pack(side="left")
    tk.Label(row, text=val,
             font=FT_MONO, bg=BG2,
             fg=val_color, anchor="w").pack(side="left")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="ANN Multi-Model Predictor GUI (Light Theme)")
    parser.add_argument(
        "--model_dir", default=None,
        help="Folder containing ann_model_1.pt ... ann_model_5.pt")
    args = parser.parse_args()

    model_dir = (pathlib.Path(args.model_dir)
                 if args.model_dir
                 else pathlib.Path(__file__).parent)

    any_found = any((model_dir / f).exists() for f in MODEL_FILES)
    if not any_found:
        root = tk.Tk(); root.withdraw()
        chosen = filedialog.askdirectory(
            title="Select folder containing ann_model_1.pt ... ann_model_5.pt")
        root.destroy()
        if not chosen:
            print("No folder selected. Exiting.")
            sys.exit(0)
        model_dir = pathlib.Path(chosen)

    app = ANNPredictorApp(model_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
