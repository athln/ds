# ds
Seismic Damage State Prediction for Corroded Low-ductility RC Frames


> A python application for predicting seismic damage states (DS-1 to DS-5) of reinforced concrete structures using Artificial Neural Network (ANN) models.

---

## Overview

This application loads pre-trained ANN models and simultaneously predicts five damage state probabilities from a set of material, geometrical and deteriorational input parameters. It is designed for structural engineers and researchers working on seismic vulnerability assessment of corrosion-affected RC structures.

**Author:** ANMK  |  NK |  NRC 
**Version:** 1.0

---

## Features

| Feature | Description |
|---|---|
| **Single Prediction** | Enter 8 structural parameters and instantly receive DS-1 to DS-5 predictions displayed as colour-coded metric cards |
| **Batch Prediction** | Upload a CSV file with multiple records; all five damage states are predicted for every row and results can be downloaded |
| **Model Info** | View architecture details, 5-fold cross-validation metrics (R², RMSE, NRMSE), and training configuration for each model |
| **Template CSV** | Download a pre-filled template CSV when no file is uploaded in batch mode |

---

## Input Parameters

All eight inputs are shared across all five models.

| # | Parameter | Unit (entered) | Internal Unit | Typical Range |
|---|---|---|---|---|
| 1 | Concrete compressive strength | MPa | MPa | 20 – 30 |
| 2 | Steel yield strength | MPa | MPa | 400 – 530 |
| 3 | Average Mass Loss | % | % | 0 – 30 |
| 4 | Pitting Mass Loss | % | % | 0 – 50 |
| 5 | Transverse Mass Loss | % | % | 0 – 70 |
| 6 | Story Height | m | m | 3.0 – 4.0 |
| 7 | Bay Width | m | m | 3.0 – 5.0 |
| 8 | Slab Thickness | m | m | 0.12 – 0.2 |

---

## Output Targets

| Output | Description |
|---|---|
| **DS-1** | Reinforcement yielding in columns/beams |
| **DS-2** | Visible crack formation in joints |
| **DS-3** | Spalling of cover concrete |
| **DS-4** | Crack width exceeding 3 mm in joints |
| **DS-5** | Shear failure in columns/beams |

---

## Model Files

Five PyTorch checkpoint files are required. Each file contains the model weights, input/output scalers (StandardScaler), training configuration, and cross-validation metrics — all packed into a single `.pt` file.

```
ann_model_1.pt   →   DS-1 predictions
ann_model_2.pt   →   DS-2 predictions
ann_model_3.pt   →   DS-3 predictions
ann_model_4.pt   →   DS-4 predictions
ann_model_5.pt   →   DS-5 predictions
```

Place all five files in the same directory as `ann_streamlit_app.py`, or specify an alternative path in the sidebar at runtime.

---

## Project Structure

```
.
├── ds-pred.py              ← Main application
├── ann_model_1.pt          ← DS-1 model checkpoint
├── ann_model_2.pt          ← DS-2 model checkpoint
├── ann_model_3.pt          ← DS-3 model checkpoint
├── ann_model_4.pt          ← DS-4 model checkpoint
├── ann_model_5.pt          ← DS-5 model checkpoint
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/athln/ds.git
cd ann-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model checkpoints

Copy your five `.pt` files into the project root:

```
ann_model_1.pt
ann_model_2.pt
ann_model_3.pt
ann_model_4.pt
ann_model_5.pt
```

---

## Running the App
# Step-1 launch GUI (auto-finds ann_model.pt in same folder)
```bash
python ds-pred.py
```
# Or point to a specific checkpoint
```bash
python ds-pred.py --model /path/to/ann_model.pt
```
To specify a custom model directory at launch


## Requirements

```
torch>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

Save as `requirements.txt` and install with `pip install -r requirements.txt`.

For CPU-only environments (no NVIDIA GPU), install the CPU build of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Usage Guide

### Single Prediction Tab

1. Adjust the 8 input sliders / number fields in the left panel.
2. Click **⚡ Predict All**.
3. Five colour-coded result cards appear on the right (DS-1 through DS-5).

### Batch Prediction Tab

1. Click **Upload input CSV** and select your file.
2. The CSV must have **8 numeric columns** in the exact order listed above.
3. The first row may be a header — it is skipped automatically.
4. After processing, a styled results table appears with all input and output columns.
5. Click **⬇ Download Full Results CSV** to save the predictions.

**CSV format example:**

```csv
Concrete Compressive Strength ,Stel Yield Strength,Average Mass Loss,Pitting Mass Loss,Transverse Mass Loss,Story Height,Bay Width,Slab Thickness
20.0,420.0,10.0,5.0,5.0,3.0,5.0,0.12
25.5,455.0,15.0,8.0,6.0,3.2,4.5,0.14
```

---

## License

This project is released for academic and research use. Please cite appropriately if used in publications.

---

## Contact

**Author:** ANMK |  NK |  NRC 
**Version:** 1.0
